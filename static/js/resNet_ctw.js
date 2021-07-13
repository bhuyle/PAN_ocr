$(document).ready(function () {
    // Init
    $('.image-section-ResNet50').hide();
    $('.loaderResNet50').hide();
    $('#resultResNet_ctw').hide();

    // Upload Preview
    function readURLResNet50(input) {
        if (input.files && input.files[0]) {
            var reader = new FileReader();
            reader.onload = function (e) {
                $('#imagePreviewResNet50').css('background-image', 'url(' + e.target.result + ')');
                $('#imagePreviewResNet50').hide();
                $('#imagePreviewResNet50').fadeIn(650);
            }
            reader.readAsDataURL(input.files[0]);
        }
    }
    $("#imageUploadResNet_pre").change(function () {
        $('.image-section-ResNet50').show();
        $('#btn-predict-ResNet_pre').show();
        $('#resultResNet_ctw').text('');
        $('#text50').text('')
        $('#resultResNet_ctw').hide();
        readURLResNet50(this);
    });

    // Predict
    $('#btn-predict-ResNet_pre').click(function () {
        var form_data = new FormData($('#upload-file-ResNet_pre')[0]);

        // Show loading animation
        $(this).hide();
        $('.loaderResNet50').show();

        // Make prediction by calling api /predictResNet50
        $.ajax({
            type: 'POST',
            url: './predictResNet_ctw',
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            async: true,
            success: function (data) {
                // Get and display the result
                $('.loaderResNet50').hide();
                $('#resultResNet_ctw').fadeIn(600);
                $("#resultResNet_ctw").attr("src","data:image/png;base64,"+ data.image);
                $('#text50').html("Number of lines "+data.box+'<br>Time '+data.time)
                // $("#resultResNet_ctw").attr("src","data:image/gif;base64," + data);
            },
        });
    });

});